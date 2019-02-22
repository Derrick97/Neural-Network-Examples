import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)
from illustrate import illustrate_results_FM


def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    model = Sequential()
    model.add(Dense(units = 25, activation = "relu", input_dim = 3))
    model.add(Dense(units = 3, activation = "softmax"))

    model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

    # input_dim = 3
    # neurons = [16, 3]
    # activations = ["relu", "identity"]
    # net = MultiLayerNetwork(input_dim, neurons, activations)
    #
    np.random.shuffle(dataset)
    x = dataset[:, :3]
    y = dataset[:, 3:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_test = x[split_idx:]
    y_test = y[split_idx:]

    model.fit(x_train, y_train, epochs = 50, batch_size = 10)
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=10)
    print(loss_and_metrics)
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
    #     loss_fun="mse",
    #     shuffle_flag=True,
    # )
    #
    # trainer.train(x_train_pre, y_train)
    # evaluate_architecture(net, trainer, x_train_pre, y_train, x_val_pre, y_val)

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    #illustrate_results_FM(model)

def evaluate_architecture(net, trainer, x_train_pre, y_train, x_val_pre, y_val):
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))

if __name__ == "__main__":
    main()
