import numpy as np
from keras.models import Sequential, load_model
from keras.layers import *
from sklearn.metrics import accuracy_score, log_loss
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.optimizers import *
import itertools


from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)

from illustrate import illustrate_results_ROI

global best_model

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

def create_model(param_combinations=(2, (["relu", "softmax"], [512, 4]))): #combination = (no. of layers, (activations, neuron_no))
    model = Sequential()
    activation_list = param_combinations[1][0]
    neuron_no_list = param_combinations[1][1]
    for i in range(param_combinations[0]):
        activation = activation_list[i]
        if i == param_combinations[0] - 1:
            neuron_no = 4
        else:
            neuron_no = neuron_no_list[i]
        if i == 0:
            model.add(Dense(units = neuron_no, activation = activation, input_dim = 3))
        else:
            #model.add(Dense(units = 50, activation = "sigmoid"))
            model.add(Dense(units = neuron_no, activation = activation))

    optimizer = Adam(lr = 0.001)
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer= optimizer,
              metrics=['accuracy'])
    return model

def main():
    dataset = np.loadtxt("ROI_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    np.random.shuffle(dataset)

    x = dataset[:, :3]
    y = dataset[:, 3:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_test = x[split_idx:]
    y_test = y[split_idx:]

    model = KerasClassifier(build_fn=create_model)



    # model = Sequential()
    # model.add(Dense(units = 512, activation = "relu", input_dim = 3))
    # model.add(Dense(units = 4, activation = "softmax"))
    #
    # optimizer = Adam(lr = 0.001)
    #
    #
    # model.compile(loss='categorical_crossentropy',
    #           optimizer= optimizer,
    #           metrics=['accuracy'])
    #
    # model.fit(x_train, y_train)
    # prediction = model.predict_proba(x_test, batch_size=10)
    # print(prediction)

    #  batch_size=10, epochs=10, param_combinations=(3, (['relu', 'relu', 'softmax'], [500, 400]))

    # Parameters
    activations = [["relu"], ["sigmoid"], ["softmax"], ["linear"]]
    neuron_no = [[300], [400], [500]]
    no_of_layers = [2, 3]
    param_combinations = generate_param_tuple(activations, neuron_no, no_of_layers)
    epochs = [10]
    batch_size = [10]

    # Create a dictionary that contains all parameters
    param_grid = dict(param_combinations=param_combinations,
                        epochs=epochs,
                        batch_size=batch_size)


    # Create the grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring="accuracy", verbose=20, cv=2)

    targets = y_train.argmax(axis=1).squeeze()

    # Run the grid search and print the accuracy of the best one
    grid_result = grid.fit(x_train, targets)
    print("Best result: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # Get the best network from grid search
    best_model = grid.best_estimator_

    best_model.model.save('my_model2.h5')

    evaluate_architecture(model, x_test, y_test)

    return best_model


    # input_dim = 3
    # neurons = [64, 4]
    # activations = ["relu", "softmax"]
    # net = MultiLayerNetwork(input_dim, neurons, activations)
    #


    # model.fit(x_train, y_train, epochs = 50, batch_size = 10)
    # # print(model.predict(np.array([[-1.570796326794896558e+00,-1.439896632895321771e+00,-1.439896632895321771e+00]])))
    #
    # model.summary()
    #
    # evaluate_architecture(model, x_test, y_test)
    #
    # prep_input = Preprocessor(x_train)
    #
    # x_train_pre = prep_input.apply(x_train)
    # x_val_pre = prep_input.apply(x_val)
    #
    # trainer = Trainer(
    #     network=net,
    #     batch_size=10,
    #     nb_epoch=100,
    #     learning_rate=0.01,
    #     loss_fun="cross_entropy",
    #     shuffle_flag=True,
    # )
    #
    # trainer.train(x_train_pre, y_train)
    # prediction = net(x_val_pre)
    # print(prediction)
    # one_hot_prediction = np.zeros_like(prediction)
    # one_hot_prediction[np.arange(len(prediction)), prediction.argmax(1)] = 1
    # evaluate_architecture(net, trainer, x_train_pre, y_train, x_val_pre, y_val, one_hot_prediction)
    # # Put the x value in validation set with the predicted y value in a ndarray
    # output = np.concatenate((x_val_pre,prediction), axis=1)
    # # Prep the data
    # prep_output = Preprocessor(output)
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # illustrate_results_ROI(net, prep_output)

def preprocess_one_hot_encoding(one_hots):
    labels = []
    for list_of_labels in one_hots:
        for index in range(len(list_of_labels)):
            if list_of_labels[index] == 1:
                labels.append(index)
    return labels

def accuracy_one_hot(y_true, y_pred):
    true = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            true += 1
    return true/len(y_true)

def cross_entropy_func(y_true, y_pred):
    ep = 1e-12
    predictions = np.clip(y_pred, ep, 1. - ep)
    sample_number = len(predictions)
    return -np.sum(np.log(predictions) * y_true)/sample_number


def evaluate_architecture(model, x_test, y_test):
    prediction = model.predict_proba(x_test, batch_size=10)
    y_test_labels = preprocess_one_hot_encoding(y_test)


    cross_entropy = cross_entropy_func(y_test, prediction)

    label_predictions = prediction.argmax(axis=1).squeeze()
    accuracy = accuracy_one_hot(y_test_labels, label_predictions)

    #loss_and_metrics = model.evaluate(x_test, y_test, batch_size=10)

    print("Test Cross Entropy = ", cross_entropy)
    print("Test accuracy: {}".format(accuracy))

def predict_hidden(new_dataset):
    x = new_dataset[:, :3]
    model = load_model('my_model2.h5')
    return model.predict(x)


# if __name__ == "__main__":
#     main()
