import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from keras import optimizers
import itertools

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)
from illustrate import illustrate_results_FM

global best_model

####################################################################
# File use to train model with params from hyper parameters search #
####################################################################


def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

def create_model(param_combinations=(1, (["relu"], [300]))): #combination = (no. of layers, (activations, neuron_no))
    model = Sequential()
    activation_list = param_combinations[1][0]
    neuron_no_list = param_combinations[1][1]
    for i in range(param_combinations[0]):
        activation = activation_list[i]
        if i == param_combinations[0] - 1:
            neuron_no = 3
        else:
            neuron_no = neuron_no_list[i]
        if i == 0:
            model.add(Dense(units = neuron_no, activation = activation, input_dim = 3))
        else:
            #model.add(Dense(units = 50, activation = "sigmoid"))
            model.add(Dense(units = neuron_no, activation = activation))

    opt = optimizers.Nadam(lr=0.01, epsilon=None, schedule_decay=0.016)
    model.compile(loss='mean_squared_error',
              optimizer= opt,
              metrics=[r_square])
    return model

def construct_model():
    dataset = np.loadtxt("FM_dataset.dat")

    np.random.shuffle(dataset)

    x = dataset[:, :3]
    y = dataset[:, 3:]

    split_idx = int(0.8 * len(x))
    #p = Preprocessor(x)
    #p.apply(x)

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_test = x[split_idx:]
    y_test = y[split_idx:]

    model = create_model((3, (["relu", "relu", "linear"], [100, 100])))

    model.fit(x_train, y_train, epochs = 100, batch_size = 10)

    model.save('model002.h5')

    evaluate_architecture(model, x_test, y_test)

def evaluate_architecture(model, x_test, y_test):
    #loss_and_metrics = model.evaluate(x_test, y_test, batch_size=10)
    prediction = model.predict(x_test, batch_size=10)

    # print(prediction)

    error = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    print("Test Mean Squared Error = ", error)
    print("Test R-Squared Value: {}".format(r2))

construct_model()
