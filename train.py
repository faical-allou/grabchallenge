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

import argparse

plt.style.use('seaborn-whitegrid')

#remove warnings tensorflow
tf.logging.set_verbosity(tf.logging.ERROR)

start_time = time.time()

###################################################   HELPER FUNCTIONS

def prep_input(data, precision_geo):
    geodata = pd.read_csv('geodata.dat')
    
    data['geo_lim'] = data.geohash6.str[0:precision_geo]
    geodata_lim = geodata.geo_id.str[0:precision_geo]

    data['time'] = pd.to_datetime(data.timestamp, format='%H:%M')
    data['date_time_index'] = data.time + pd.TimedeltaIndex(data.day, unit = 'D')
    
    datagg = data.groupby(['geo_lim','date_time_index'])['demand'].mean().reset_index(name='mean')
    Xagg = datagg.pivot(index='date_time_index', columns='geo_lim', values='mean').fillna(0, inplace=False)

    for c_lim in geodata_lim:
        if c_lim not in Xagg : Xagg[c_lim] = 0
    
    Xfull = data.pivot(index='date_time_index', columns='geohash6', values='demand').fillna(0, inplace=False)
    for c in geodata.geo_id:
        if c not in Xfull : Xfull[c] = 0

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
    return np.array(out).clip(min=0, max=1)

def scaling(Y,scaling_vector):
    scaling_vector[scaling_vector>3] = 1
    out = Y*scaling_vector  
    return np.array(out)

def sum_pred_error(y_true, y_pred):
    return K.abs(K.mean(y_pred)-K.mean(y_true))+K.mean(K.abs(y_pred-y_true)**2)


###################################################   PRINTING FUNCTIONS


def graphit(Yt,Yh,index,title):
    plt.figure(index)
    plt.plot(Yt.flatten(), Yh.flatten(), 'o', color='black')
    plt.plot([0,1],[0,1])
    plt.xlabel("actual")
    plt.ylabel("prediction")
    plt.suptitle(title)
    plt.show(block=False)
    input('press <ENTER> to continue')

def output_csv(filename, Yt, Yh, column_name, rows_to_print ):
    myFile = open(filename, 'w', newline='')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerow(column_name)
        for i in range(rows_to_print):
            writer.writerow((100*Yt[i]).astype(int))
            writer.writerow((100*Yh[i]).astype(int))
            writer.writerow(" ")
        writer.writerow(" ##### END ##### ")

def KPI(Yt, Yh, index):
    mae = (np.abs(Yt-Yh)).mean(axis=None)
    mape = np.sum(np.abs(Yt-Yh))/np.sum(Yt)
    mse = ((Yt-Yh)**2).mean(axis=None)

    print("step "+str(index)+" test MAE: %.2f" % mae)
    print("step "+str(index)+" test MAPE: %.2f%%" % (mape*100))
    print("step "+str(index)+" test MSE: %.2f%%" % (mse*100))
    print(" ") 
    return mae, mape, mse

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

def execute_script(filename, train1 = False, e=100, e2=1000, train2=False, rescale=False):
###################################################   INPUT
    data = pd.read_csv(filename)
    print("reading data done : "+str(int(time.time()-start_time))+" s")

    test_periods = 5 # following the test period
    precision = 5 # number of digit in the geo param (max is 6)  this parameter increases size O(36^n)
    lookback = 4*4 # number of periods to lookback; 4 per hour 

    start_from_previous = True  # if you train do you start from the previous

    start_from_previous2 = True # if you train2 do you start from the previous

    p = 8           # number of periods to average for the scaling starting from the last training period
    scaling_vector = np.genfromtxt('scaling.dat', delimiter=',')

    ##################################################  PREP INPUT
    Xprep, Xfull = prep_input(data, precision)
    training_periods = len(Xprep.index)-test_periods-2

    ##################################################  HYPERPARAMETERS
    nbcolumns = len(Xprep.columns)
    h_layer1_nodes = int(nbcolumns*lookback)
    h_layeri_nodes = int(nbcolumns*lookback)
    h_layerf_nodes = int(nbcolumns*lookback)

    b = int(training_periods/2)     # batch size

    ###################################################   DATA PREP FOR NETWORK 1

    Xprepn, mXprepn, sXprepn = normalize(Xprep)

    #remove first time step from Y and the last from X 
    Yprepn = Xprepn[1:]
    Xprepn.drop(Xprepn.tail(1).index,inplace=True)

    # filtering for the dates we need
    Xn = np.array(Xprepn[lookback:training_periods])
    Yn = np.array(Yprepn[lookback:training_periods])

    #shaping for the LSTM layer requirement
    Xn = Xn.reshape(Xn.shape[0],1,Xn.shape[1])

    # adding the data to look back at
    for j in range(lookback-1):
        Xlookback = np.array(Xprepn[lookback-j-1:training_periods-j-1])
        Xlookback = Xlookback.reshape(Xlookback.shape[0],1,Xlookback.shape[1])
        Xn = np.append( Xn, Xlookback , axis=1 )

    # preparing the test data
    Xtest = np.array(Xprepn[training_periods:training_periods+test_periods])
    Xtest = Xtest.reshape(Xtest.shape[0],1,Xtest.shape[1])

    # adding the data to look back at
    for j in range(lookback-1):
        Xlookback = np.array(Xprepn[training_periods-j-1:training_periods+test_periods-j-1])
        Xlookback = Xlookback.reshape(Xlookback.shape[0],1,Xlookback.shape[1])
        Xtest = np.append( Xtest, Xlookback , axis=1 )

    Ytest = np.array(Xprep[training_periods+1:training_periods+test_periods+1]) # using non normalized test output

    print("data prep done : " +str(int(time.time()-start_time))+" s")

    ###################################################   MODEL 1
    print("modelling start: " +str(int(time.time()-start_time))+" s")
    # create model
    model = Sequential()
    model.add(LSTM(h_layer1_nodes, input_shape=(lookback, nbcolumns), activation='tanh', return_sequences=True))  
    model.add(LSTM(h_layerf_nodes, activation='tanh'))
    model.add(Dense(nbcolumns, activation='linear')) #output layer

    if start_from_previous: model.load_weights("model.h5x")

    if train1:
        model.compile(loss='mean_absolute_error', optimizer='sgd', metrics=['mean_absolute_error'])
        model.fit(Xn, Yn, epochs=e, batch_size=b, validation_split=0.2,  verbose=2)

        scores = model.evaluate(Xn, Yn)
        model.save_weights("model.h5x")
        print(" ")
        print("training1 in : "+str(int(time.time()-start_time))+" s")
        print(" ")

    else:
        model.load_weights("model.h5x")

    ####################################################  SCALING FACTORS
    Yscalen = model.predict(Xn)
    if rescale:
        scaling_vector = np.sum(Yn[-p:], axis=0)/np.sum(Yscalen[-p:], axis=0)

        myFile = open('scaling.dat', 'w', newline='')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow(scaling_vector)

    print("scaling done: "+str(int(time.time()-start_time))+" s")
    ###################################################   DATA PREP FOR NETWORK 2

    full_size = len(Xfull.columns)

    Y2 = np.array(Xfull[lookback+1:training_periods+1])

    Ytest2 = np.array(Xfull[training_periods+2:training_periods+test_periods+2])

    print("Prep 2 done : "+str(int(time.time()-start_time))+" s")

    ###################################################   MODEL 2

    model2 = Sequential()
    model2.add(Dense(full_size, activation='sigmoid', input_dim=nbcolumns,bias_regularizer=regularizers.l1(0.1)))
    model2.add(Dense(full_size, activation='sigmoid', input_dim=nbcolumns,bias_regularizer=regularizers.l1(0.1)))

    if start_from_previous2: model2.load_weights("model2.h5x")

    if train2:

        fitted_m1s = scaling(Yscalen, scaling_vector)
        fitted_m1 = denormalize(fitted_m1s, mXprepn, sXprepn)

        model2.compile(loss=sum_pred_error, optimizer='adam', metrics=['mean_absolute_error'])

        model2.fit(fitted_m1, Y2, epochs=e2, batch_size=b, validation_split=0.2,  verbose=2)
        model2.save_weights("model2.h5x")

        scores = model2.evaluate(fitted_m1, Y2)
        print(" ")
        print("training2 in : "+str(int(time.time()-start_time))+" s")
        print(" ")

    else: 
        model2.load_weights("model2.h5x")

    ###################################################   PREDICT / TIME_SCALE / EXPLODE 

    Yhatn = model.predict(Xtest)
    Yhats = scaling(Yhatn, scaling_vector)
    Yhat = denormalize(Yhats, mXprepn, sXprepn)
    Yhat2 = model2.predict(Yhat)

    print("all predictions done : "+str(int(time.time()-start_time))+" s\n")
    ###################################################   RESULTS

    ####output of first model
    if train1 : 
        print_weights(model)

        mae, mape, mse = KPI(Ytest, Yhat, 1)
        graphit(Ytest, Yhat, 0, "Aggregate")
        output_csv('output.csv', Ytest, Yhat, Xprepn.columns, test_periods )
            
        model_name = "./models/model "+"UTC "+str(datetime.now())+" "+str(training_periods)+" MAPE="+str(mape)+" MSE= "+str(mse)+".h5"
        model_name= re.sub('[:]+', '', model_name)
        model.save(model_name)

    ####output of second model
    if train2: 
        print_weights(model2)

        KPI(Ytest2, Yhat2, 2)
        graphit(Ytest2, Yhat2, 1, "Granular")
        output_csv('output2.csv', Ytest2, Yhat2, Xprepn.columns, test_periods )
        
    print("end of script in : "+str(int(time.time()-start_time))+" s")
    return Xprepn, mXprepn, sXprepn, Xfull, full_size, nbcolumns, test_periods, lookback, model, model2, scaling_vector

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-i","--inputfile", help="name of the input file, default = input.csv",type=str)
    parser.add_argument("-t1","--train1", help="train first network, default = False",action="store_true")
    parser.add_argument("-e1","--epochs1", help="epochs to train the first network, default = 100", type=int)
    parser.add_argument("-rs","--rescale", help="recalculate the scaling of the output of first network, default = False",action="store_true")
    parser.add_argument("-t2","--train2", help="train second network, default = False",action="store_true")
    parser.add_argument("-e2","--epochs2", help="epochs to train the second network, default = 1000", type=int)
 
    parser.add_argument("-f","--full", help="set all training paramters to true, default = False",action="store_true")
 
    args = parser.parse_args()

    filename = args.inputfile if args.inputfile else 'input.csv'

    epoch1 = args.epochs1 if args.epochs1 else  100
    epoch2 = args.epochs2 if args.epochs2 else  3000
    
    if args.full :
        args.train1 = True
        args.train2 = True
        args.rescale = True
        

    execute_script(filename,train1 = args.train1, e=epoch1, train2=args.train2, e2=epoch2, rescale=args.rescale)