import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)

from illustrate import illustrate_results_ROI


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
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    model = Sequential()
    model.add(Dense(units = 512, activation = "relu", input_dim = 3))
    model.add(Dense(units = 4, activation = "softmax"))
    optimizer = Adam(lr = 0.001)

    model.compile(loss='categorical_crossentropy',
              optimizer= optimizer,
              metrics=['accuracy'])



    # input_dim = 3
    # neurons = [64, 4]
    # activations = ["relu", "softmax"]
    # net = MultiLayerNetwork(input_dim, neurons, activations)
    #


    model.fit(x_train, y_train, epochs = 50, batch_size = 10)

    model.summary()
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

def evaluate_architecture(net, trainer, x_train_pre, y_train, x_val_pre, y_val, prediction):
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = prediction.argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))

if __name__ == "__main__":
    main()
