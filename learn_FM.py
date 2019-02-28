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
    neuron_no = [[25], [50], [75], [100]]
    no_of_layers = [2, 3, 4]
    param_combinations = generate_param_tuple(activations, neuron_no, no_of_layers)
    epochs = [5]
    batch_size = [10, 20]

    # Create a dictionary that contains all parameters
    param_grid = dict(param_combinations=param_combinations,
                        epochs=epochs,
                        batch_size=batch_size)

    # Create the random search
    # grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring="r2", verbose=20)

    # Create the grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring="r2", verbose=20)

    # input_dim = 3
    # neurons = [16, 3]
    # activations = ["relu", "identity"]
    # net = MultiLayerNetwork(input_dim, neurons, activations)
    #
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

    # Run the grid search and print the accuracy of the best one
    grid_result = grid.fit(x_train, y_train)
    print("Best result: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # Get the best network from grid search
    best_model = grid.best_estimator_

    best_model.model.save('my_model.h5')

    #model.fit(x_train, y_train, epochs = 50, batch_size = 10)
    #y_pred = model.predict(np.array([[1.570796326794896558e+00,1.439896632895321549e+00,-5.235987755982989267e-01]]))#,4.684735272501165284e-15,1.094144784178339336e+02,6.310930546237394765e+02]]))
    #y_ = y_test[900:]
    #print(y_pred[0])
    #print(y_[0])

    evaluate_architecture(best_model, x_test, y_test)
    #save_network(model, "/homes/jr2216/neuralnetworks_34/model.dat")

    return best_model
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    #illustrate_results_FM(model)



def predict_hidden(new_dataset):
    #model = load_network("/homes/jr2216/neuralnetworks_34/model.dat")
    x = new_dataset[:, :3]
    model = load_model('my_model.h5', custom_objects={'r_square': r_square})
    return model.predict(x)


def evaluate_architecture(model, x_test, y_test):
    #loss_and_metrics = model.evaluate(x_test, y_test, batch_size=10)
    prediction = model.predict(x_test, batch_size=10)

    # print(prediction)

    error = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    print("Test Mean Squared Error = ", error)
    print("Test R-Squared Value: {}".format(r2))

model = construct_model()

# if __name__ == "__main__":
#     main()
print(predict_hidden(np.array([[1.570796326794896558e+00,1.439896632895321549e+00,-5.235987755982989267e-01,4.684735272501165284e-15,1.094144784178339336e+02,6.310930546237394765e+02]])))

# print(generate_param_tuple([["relu"], ["sigmoid"], ["linear"]], [[50], [100]], [1,2,3]))
