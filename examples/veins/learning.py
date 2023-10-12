import os
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from python.src import constants, models, logs, gossip_learning

vehicle_data = {}
vehicle_models = {}

def init(car_id, sim_time):
    for file in os.listdir(constants.DATA_PATH):
        if file.startswith(car_id + '_'):
            logging.warning('Preparing dataset for ' + file)
            data = np.load(constants.DATA_PATH + file)
            X, y = data['images_train'], data['labels_train']
            num_classes = data['num_classes']

            skf = StratifiedKFold(n_splits=5)
            for i, (train_index, valid_index) in enumerate(skf.split(X, y)):
                if constants.SPLIT == i:
                    X_train, y_train = X[train_index], y[train_index]
                    X_valid, y_valid = X[valid_index], y[valid_index]
                    vehicle_data[car_id] = {}
                    vehicle_data[car_id]['train'] = (X_train, keras.utils.to_categorical(y_train, num_classes))
                    vehicle_data[car_id]['valid'] = (X_valid, keras.utils.to_categorical(y_valid, num_classes))
            logging.warning('Dataset preparation finished')

            vehicle_models[car_id] = models.get_model(num_classes)

            logs_data = {'event': 'init', 'car_id': car_id, 'sim_time': sim_time}
            logs.register_log(logs_data)


def train(car_id, training_round, sim_time):
    if constants.EXPERIMENT == constants.GOSSIP_LEARNING:
        gossip_learning.train(car_id, training_round, sim_time, vehicle_data, vehicle_models)

def get_weights(car_id, sim_time):
    logs_data = {'event': 'get_weights', 'car_id': car_id, 'sim_time': sim_time}
    logs.register_log(logs_data)

    if constants.EXPERIMENT == constants.GOSSIP_LEARNING:
        gossip_learning.get_weights(car_id, vehicle_models)

def merge(raw_weights, car_id, sender_id, sim_time):
    logging.warning('Node {} merging models'.format(car_id))
    logs_data = {'event': 'merge', 'car_id': car_id, 'sim_time': sim_time, 'sender_id': sender_id}
    logs.register_log(logs_data)

    if constants.EXPERIMENT == constants.GOSSIP_LEARNING:
        gossip_learning.merge(raw_weights, car_id, vehicle_models)
