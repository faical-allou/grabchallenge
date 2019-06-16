# grabchallenge

This is a submission for the challenge "Traffic Management" published by grab in May 2019
The website is here : https://www.aiforsea.com/traffic-management

The task consist in predicting the demand for rides for 5 time steps (of 15 min) given data for the preceding 2 weeks.

##########

The input data comes in a csv file, with the following columns: "geohash6","day","timestamp","demand", the file should be comma (",") separated

The script to run are "train.py -f" and "predict.py"

Both come with options that can be passed from the command line, and -h will list them.

This submision should be judged on the basis of the following step by step:

1) install the dependencies:
pip install -r requirements.txt

2) retrain the networks (note the option -f that means to retrain the entire set with default values)
python train.py -f

3) predict the next 5 timesteps
python predict.py 

This will generate a file called "prediction.csv"

############

The solution offered consist in several steps:

a) aggregate the data one level higher (geohash5) and normalize it
b) predict the next time steps at the aggregate normalized level with a recurrent neural network (2 hidden LSTM layers)
c) denormalize and scale the data to avoid topline demand loss (most recent training period vs actual)
d) predict the time step at the granular level (geohash6) by using the forecast of the step before with a "vanilla" neural network (1 hidden layers)

###########

There are several options available from the command line:
for train.py
-t1 will only train the recurrent network
-e1 <value> will train the recurrent network for the given number of epochs (default is 100)

-t2 will only train the second network
-e2 <value> will train the scond network for the given number of epochs(default is 3000)



