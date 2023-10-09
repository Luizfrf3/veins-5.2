import os
import logging
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras import layers

SEED = 12
SPLIT = 0
EPOCHS = 1
keras.utils.set_random_seed(SEED)

GOSSIP_LEARNING = 'Gossip Learning'
OUR_METHOD = 'Our Method'
FED_AVG = 'FedAvg'
FED_PROX = 'FedProx'
WSCC = 'WSCC'
FEDPC = 'FedPC'
BARBIERI = 'Barbieri'
DFL_DSS = 'DFL-DSS'
EXPERIMENT = GOSSIP_LEARNING

MNIST = 'MNIST'
CIFAR10 = 'CIFAR10'
FEMNIST = 'FEMNIST'
GTSRB = 'GTSRB'
DATASET = MNIST

DATA_PATH = 'python/' + DATASET + '/data/'
WEIGHTS_FOLDER = 'weights/'
WEIGHTS_FILE_SUFFIX = '_weights.pickle'

vehicle_data = {}
models = {}

def _get_model(num_classes):
    model = None

    if DATASET == MNIST:
        model = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Dropout(0.25),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax")
            ]
        )
    elif DATASET == CIFAR10:
        model = keras.Sequential(
            [
                keras.Input(shape=(32, 32, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Dropout(0.25),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax")
            ]
        )
    elif DATASET == FEMNIST:
        model = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(32, kernel_size=(5, 5), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                layers.Conv2D(64, kernel_size=(5, 5), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                layers.Flatten(),
                layers.Dense(2048, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax")
            ]
        )
    elif DATASET == GTSRB:
        model = keras.Sequential(
            [
                keras.Input(shape=(48, 48, 3)),
                layers.Conv2D(1, kernel_size=(1, 1), padding="same", activation="relu"),
                layers.Conv2D(29, kernel_size=(5, 5), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                layers.Conv2D(59, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                layers.Conv2D(74, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                layers.Flatten(),
                layers.Dense(300, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax")
            ]
        )

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def _save_weights(weights_path, weights):
    with open(weights_path, 'wb') as weights_file:
        pickle.dump(weights, weights_file)

def _encode_weights(weights):
    weights_bytes = pickle.dumps(weights)
    raw_weights = ''
    for byte in weights_bytes:
        raw_weights += str(byte) + ','
    return raw_weights[:-1]

def _decode_weights(raw_weights):
    byte_list = []
    for byte_str in raw_weights.split(','):
        byte_list.append(int(byte_str))
    weights_bytes = bytes(byte_list)
    return pickle.loads(weights_bytes)

def init(car_id):
    for file in os.listdir(DATA_PATH):
        if file.startswith(car_id + '_'):
            logging.warning('Preparing dataset for ' + file)
            data = np.load(DATA_PATH + file)
            X, y = data['images_train'], data['labels_train']
            num_classes = data['num_classes']

            skf = StratifiedKFold(n_splits=5)
            for i, (train_index, valid_index) in enumerate(skf.split(X, y)):
                if SPLIT == i:
                    X_train, y_train = X[train_index], y[train_index]
                    X_valid, y_valid = X[valid_index], y[valid_index]
                    vehicle_data[car_id] = {}
                    vehicle_data[car_id]['train'] = (X_train, keras.utils.to_categorical(y_train, num_classes))
                    vehicle_data[car_id]['valid'] = (X_valid, keras.utils.to_categorical(y_valid, num_classes))
            logging.warning('Dataset preparation finished')

            models[car_id] = _get_model(num_classes)

def train(car_id, training_round):
    X_train, y_train = vehicle_data[car_id]['train']
    X_valid, y_valid = vehicle_data[car_id]['valid']

    model = models[car_id]
    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_valid, y_valid), verbose=0)
    logging.warning('Node {}, Training Round {}, History {}'.format(car_id, training_round, history.history))

    weights_path = WEIGHTS_FOLDER + car_id + WEIGHTS_FILE_SUFFIX
    _save_weights(weights_path, model.get_weights())

def get_weights(car_id):
    model = models[car_id]
    return _encode_weights(model.get_weights())

def merge(raw_weights, car_id):
    logging.warning('Node {} merging models'.format(car_id))
    model = models[car_id]
    received_weights = _decode_weights(raw_weights)
    weights = model.get_weights()
    for i in range(len(weights)):
        weights[i] = (weights[i] + received_weights[i]) / 2
    model.set_weights(weights)
