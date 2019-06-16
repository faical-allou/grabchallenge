# grabchallenge

This is a submission for the challenge "Traffic Management" published by grab in May 2019.

The website is here : https://www.aiforsea.com/traffic-management

The task consist in predicting the demand for rides for 5 time steps (of 15 min) given data for the preceding 2 weeks.

-------

The input data comes in a csv file, with the following columns: "geohash6","day","timestamp","demand", the file should be comma (",") separated

This submision should be judged on the basis of the following step by step, assuming the input file is called **input.csv**:

1) install the dependencies: 

`pip install -r requirements.txt`

2) retrain the networks for the duration given (note the option -f that means to retrain the entire set with default values):

`python train.py -f`

3) predict the next 5 timesteps:

`python predict.py `

This will generate a file called **prediction.csv**

------

The solution submitted here consists of several steps:

* aggregate the data one level higher (geohash5) and normalize it
* predict the next time steps at the aggregate normalized level with a recurrent neural network (2 hidden LSTM layers)
* denormalize and scale the data to avoid topline demand loss (most recent training period vs actual)
* predict the time step at the granular level (geohash6) by using the forecast of the step before with a "vanilla" neural network (1 hidden layers)

------

There are several options available from the command line:

``` 
-h for help 
-i to name the input file unless it is called "input.csv"
```

for **train.py**
```
-t1 will only train the recurrent network
-e1 <value> will train the recurrent network for the given number of epochs (default is 100)

-t2 will only train the second network
-e2 <value> will train the second network for the given number of epochs (default is 3000)

-rs with recalculate the scaling factor

-f will retrain all based on the default values

```
-----
for **predict.py**

```
-o to name the output file otherwise it is called "prediction.csv"
```

----
# Note:

If `python train.py -f` takes too long you can consider running `python train.py -t2` and only train the second network which is where most of the training benefits are.

The file **geodata.dat** contains a list of the geohash6 ids found in the initial training data. This enables to use data that contains less geohash6 id, but not more. The code will fail if an additional geohash6 data is introduced. In this case, the network will need to be reset by changing the variable *start_from_previous* and *start_from_previous2* to *False* in the **train.py** file.

The file **scaling_vector.dat** contains the correction factor that is applied after the output of the first network and before entering the second one.

The files **model.h5x** and **model2.h5x** are "Keras exports" of the weights to be used as pretraining for both networks.

After training each network will replace the **.h5x** files. In addition the first network will generate a **model.h5** file to be saved in the *models* folder.



