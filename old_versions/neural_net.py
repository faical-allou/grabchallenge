import numpy as np
import csv


# sigmoid function
def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

##### start here

data = np.genfromtxt('subdata.csv', delimiter=',', dtype=str)



def prep_input(data, day):
    Xprep = np.delete(data, (0), axis=0)
    Xprep = Xprep[Xprep[:,1] <= str(day) ,:]
    list_geo = np.unique(Xprep[:,0])
    weekday = np.remainder(Xprep[:,1].astype(int),7)

    onehot_geo, keys_geo = onehot_encoding(Xprep[:,0])
    onehot_day, keys_day = onehot_encoding(Xprep[:,1])
    onehot_weekday, keys_weekday = onehot_encoding(weekday)
    onehot_time, keys_time = onehot_encoding(Xprep[:,2])
    
    X = np.hstack((onehot_geo, onehot_day, onehot_weekday, onehot_time))
    return X

def prep_label(data,day):
    Yprep = np.delete(data, (0), axis=0)
    Yprep = Yprep[Yprep[:,1] <= str(day),:]
    Y = Yprep[:,3].astype(float)
    Y = Y.reshape((-1,1))
    return Y


def onehot_encoding(vector_to_convert):
    
    unique_values = np.unique(vector_to_convert)
    onehot = np.zeros((1,unique_values.size))
    for v in vector_to_convert:
        onehot_row = np.array(v == unique_values).astype(int)
        onehot = np.vstack((onehot, onehot_row))
    onehot = np.delete(onehot, (0), axis=0)

    return onehot, unique_values


############################

X = prep_input(data,10)
Y = prep_label(data, 10)

input_layer_size = X[0,:].size

np.random.seed(1)
syn0 = 2*np.random.random((input_layer_size,1)) - 1

for iter in range(10000):

    # forward propagation
    l0 = X
    l1 = sigmoid(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = Y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * sigmoid(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print("Output After Training:")
print(np.sum(l1-Y))

