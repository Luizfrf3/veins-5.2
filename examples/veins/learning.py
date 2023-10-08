import os
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras

SEED = 12
SPLIT = 0
tf.keras.utils.set_random_seed(SEED)

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

vehicle_data = {}

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
