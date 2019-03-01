import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from keras import optimizers
import itertools
import sys

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)
from illustrate import illustrate_results_FM

global best_model

# Function for calculaing the r-square value
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

# Function for creating the model
def create_model(param_combinations=(1, (["relu"], [300]))): #combination = (no. of layers, (activations, neuron_no))
    model = Sequential()
    # Get thn list of activation function and neuron number from the tuple
    activation_list = param_combinations[1][0]
    neuron_no_list = param_combinations[1][1]
    # For loop for adding all layers
    for i in range(param_combinations[0]):
        # Get the activation function for this layer
        activation = activation_list[i]
        if i == param_combinations[0] - 1: # If this is the last layer, the number of neuron is 3
            neuron_no = 3
        else: # Otherwise, get the number of neuron for this layer from the list
            neuron_no = neuron_no_list[i]
        if i == 0: # If it is the first layer, specify input_dim = 3
            model.add(Dense(units = neuron_no, activation = activation, input_dim = 3))
        else:
            model.add(Dense(units = neuron_no, activation = activation))

    # Compile the model
    model.compile(loss='mean_squared_error',
              optimizer= 'Nadam',
              metrics=[r_square])
    return model

# This function generates a list of all combination of
# (number of layers, (activation function, number of neurons)) (int, (list, list))
def generate_param_tuple(activations, neuron_no, no_of_layers):
    result = []
    for n in no_of_layers:
        ac_list = activations
        neu_no_list = neuron_no
        for i in range(n - 1):
            ac_list = list(itertools.product(ac_list, activations))
            ac_list = [a + b for (a, b) in ac_list]
            if i != n - 2:
                neu_no_list = list(itertools.product(neu_no_list, neuron_no))
                neu_no_list = [a + b for (a, b) in neu_no_list]
        if n == 1:
            neu_no_list = [[]]
        final_list = list(itertools.product(ac_list, neu_no_list))
        final_list = [(n, x) for x in final_list]
        result = result + final_list
    return result



def construct_model():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # Model for hyperparameter search
    model = KerasRegressor(build_fn=create_model)

    # Parameters
    activations = [["relu"], ["sigmoid"], ["linear"]]
    neuron_no = [[25], [50]]
    no_of_layers = [3]
    param_combinations = generate_param_tuple(activations, neuron_no, no_of_layers)
    epochs = [5]
    batch_size = [10, 20, 30]

    # Create a dictionary that contains all parameters
    param_grid = dict(param_combinations=param_combinations,
                        epochs=epochs,
                        batch_size=batch_size)


    # Create the grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring="r2", verbose=20)

    # Shuffle the dataset
    np.random.shuffle(dataset)

    # Split the data into x and y
    x = dataset[:, :3]
    y = dataset[:, 3:]

    # Index of spliting the training ant testing set
    split_idx = int(0.8 * len(x))

    # Spli x and y into training ant testing set
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_test = x[split_idx:]
    y_test = y[split_idx:]

    # Run the grid search and print the accuracy of the best one
    grid_result = grid.fit(x_train, y_train)
    print("Best result: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # Get the best network from grid search
    best_model = grid.best_estimator_

    # Save the best model
    best_model.model.save('my_model.h5')

    evaluate_architecture(best_model, x_test, y_test)

    return best_model
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    #illustrate_results_FM(model)



def predict_hidden(new_dataset):
    # Get the x for the new dtaset
    x = new_dataset[:, :3]
    # Load the best model trained
    model = load_model('model001.h5', custom_objects={'r_square': r_square})
    # Return the prediction
    return model.predict(x)


def evaluate_architecture(model, x_test, y_test):
    # Get the predicionn of the model
    prediction = model.predict(x_test, batch_size=10)

    # Calculate the mean-square error and the r2 score
    error = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    # Print the result
    print("Test Mean Squared Error = ", error)
    print("Test R-Squared Value: {}".format(r2))

# model = construct_model()

# print(predict_hidden(np.array([[1.570796326794896558e+00,1.439896632895321549e+00,-5.235987755982989267e-01,4.684735272501165284e-15,1.094144784178339336e+02,6.310930546237394765e+02]])))
filename = sys.argv[1]
new_dataset = np.loadtxt(filename)
print(predict_hidden(new_dataset))
