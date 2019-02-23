import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

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


def construct_model():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    model = Sequential()
    model.add(Dense(units = 300, activation = "relu", input_dim = 3))
    #model.add(Dense(units = 50, activation = "sigmoid"))
    model.add(Dense(units = 3, activation = "linear"))

    sgd = optimizers.SGD(lr = 0.01, decay = 0.0, clipnorm = 1);

    model.compile(loss='mean_squared_error',
              optimizer= 'Nadam',
              metrics=[r_square])

    # input_dim = 3
    # neurons = [16, 3]
    # activations = ["relu", "identity"]
    # net = MultiLayerNetwork(input_dim, neurons, activations)
    #
    np.random.shuffle(dataset)

    x = dataset[:, :3]
    y = dataset[:, 3:]

    split_idx = int(0.8 * len(x))
    p = Preprocessor(x)
    p.apply(x)

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_test = x[split_idx:]
    y_test = y[split_idx:]

    model.fit(x_train, y_train, epochs = 50, batch_size = 10)
    #y_pred = model.predict(np.array([[1.570796326794896558e+00,1.439896632895321549e+00,-5.235987755982989267e-01]]))#,4.684735272501165284e-15,1.094144784178339336e+02,6.310930546237394765e+02]]))
    # y_ = y_test[900:]
    #print(y_pred[0])
    # print(y_[0])

    evaluate_architecture(model, x_test, y_test)
    #save_network(model, "/homes/jr2216/neuralnetworks_34/model.dat")

    return model
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    #illustrate_results_FM(model)



def predict_hidden(new_dataset):
    #model = load_network("/homes/jr2216/neuralnetworks_34/model.dat")
    x = new_dataset[:, :3]
    return model.predict(x)


def evaluate_architecture(model, x_test, y_test):
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=10)

    print("Test Mean Squared Error = ", loss_and_metrics[0])
    print("Test R-Squared Value: {}".format(loss_and_metrics[1]))

model = construct_model()

if __name__ == "__main__":
    main()
    #predict_hidden(np.array([[1.570796326794896558e+00,1.439896632895321549e+00,-5.235987755982989267e-01,4.684735272501165284e-15,1.094144784178339336e+02,6.310930546237394765e+02]]))
