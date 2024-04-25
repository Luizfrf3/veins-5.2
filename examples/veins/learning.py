import os
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from python.src import constants, models, logs, gossip_learning, our_method, fed_avg_fed_prox, wscc

vehicle_data = {}
node_models = {}

def init(node_id, sim_time):
    for file in os.listdir(constants.DATA_PATH):
        if file.startswith(node_id + '_'):
            logging.warning('Preparing dataset for ' + file)
            data = np.load(constants.DATA_PATH + file)
            X, y = data['images_train'], data['labels_train']
            num_classes = data['num_classes']

            skf = StratifiedKFold(n_splits=5)
            for i, (train_index, valid_index) in enumerate(skf.split(X, y)):
                if constants.SPLIT == i:
                    X_train, y_train = X[train_index], y[train_index]
                    X_valid, y_valid = X[valid_index], y[valid_index]
                    vehicle_data[node_id] = {}
                    vehicle_data[node_id]['train'] = (X_train, keras.utils.to_categorical(y_train, num_classes))
                    vehicle_data[node_id]['valid'] = (X_valid, keras.utils.to_categorical(y_valid, num_classes))

            np.savez(
                constants.DATA_GENERATED_FOLDER + file,
                X_valid=vehicle_data[node_id]['valid'][0],
                y_valid=vehicle_data[node_id]['valid'][1]
            )
            logging.warning('Dataset preparation finished')

            node_models[node_id] = models.get_model()

            logs_data = {'event': 'init', 'node_id': node_id, 'sim_time': sim_time}
            logs.register_log(logs_data)

def init_server(node_id, sim_time):
    node_models[node_id] = models.get_model()

    logs_data = {'event': 'init_server', 'node_id': node_id, 'sim_time': sim_time}
    logs.register_log(logs_data)

def train(node_id, training_round, sim_time):
    if constants.EXPERIMENT == constants.GOSSIP_LEARNING:
        gossip_learning.train(node_id, training_round, sim_time, vehicle_data, node_models)
    elif constants.EXPERIMENT == constants.OUR_METHOD:
        our_method.train(node_id, training_round, sim_time, vehicle_data, node_models)
    elif constants.EXPERIMENT == constants.FED_AVG:
        fed_avg_fed_prox.train(node_id, training_round, sim_time, vehicle_data, node_models)
    elif constants.EXPERIMENT == constants.FED_PROX:
        fed_avg_fed_prox.train(node_id, training_round, sim_time, vehicle_data, node_models)
    elif constants.EXPERIMENT == constants.WSCC:
        wscc.train(node_id, training_round, sim_time, vehicle_data, node_models)

def get_weights(node_id, sim_time):
    logs_data = {'event': 'get_weights', 'node_id': node_id, 'sim_time': sim_time}
    logs.register_log(logs_data)

    model = node_models[node_id]
    return models.encode_weights(model.get_weights())

def get_dataset_size(node_id):
    return len(vehicle_data[node_id]['train'][0])

def merge(raw_weights, dataset_size, node_id, sender_id, sim_time):
    logging.warning('Node {} merging models'.format(node_id))
    logs_data = {'event': 'merge', 'dataset_size': dataset_size, 'node_id': node_id, 'sim_time': sim_time, 'sender_id': sender_id}
    logs.register_log(logs_data)

    if constants.EXPERIMENT == constants.GOSSIP_LEARNING:
        gossip_learning.merge(raw_weights, dataset_size, node_id, vehicle_data, node_models)

def store_weights(raw_weights, dataset_size, node_id, sender_id, sim_time):
    logs_data = {'event': 'store_weights', 'dataset_size': dataset_size, 'node_id': node_id, 'sim_time': sim_time, 'sender_id': sender_id}
    logs.register_log(logs_data)

    if constants.EXPERIMENT == constants.OUR_METHOD:
        our_method.store_weights(raw_weights, dataset_size, node_id, sender_id)
    elif constants.EXPERIMENT == constants.FED_AVG:
        fed_avg_fed_prox.store_weights(raw_weights, dataset_size, node_id, sender_id)
    elif constants.EXPERIMENT == constants.FED_PROX:
        fed_avg_fed_prox.store_weights(raw_weights, dataset_size, node_id, sender_id)
    elif constants.EXPERIMENT == constants.WSCC:
        wscc.store_weights(raw_weights, dataset_size, node_id, sender_id)

def store_weights_while_training(raw_weights, dataset_size, node_id, sender_id, sim_time):
    logs_data = {'event': 'store_weights_while_training', 'dataset_size': dataset_size, 'node_id': node_id, 'sim_time': sim_time, 'sender_id': sender_id}
    logs.register_log(logs_data)

    if constants.EXPERIMENT == constants.OUR_METHOD:
        our_method.store_weights_while_training(raw_weights, dataset_size, node_id, sender_id)

def receive_global_model(raw_weights, node_id, sender_id, sim_time):
    if constants.EXPERIMENT == constants.FED_AVG:
        fed_avg_fed_prox.receive_global_model(raw_weights, node_id, sender_id, sim_time, node_models)
    elif constants.EXPERIMENT == constants.FED_PROX:
        fed_avg_fed_prox.receive_global_model(raw_weights, node_id, sender_id, sim_time, node_models)
    elif constants.EXPERIMENT == constants.WSCC:
        wscc.receive_global_model(raw_weights, node_id, sender_id, sim_time, node_models, vehicle_data)

def aggregation(aggregation_round, node_id, sim_time):
    if constants.EXPERIMENT == constants.FED_AVG:
        fed_avg_fed_prox.aggregation(aggregation_round, node_id, sim_time, node_models)
    elif constants.EXPERIMENT == constants.FED_PROX:
        fed_avg_fed_prox.aggregation(aggregation_round, node_id, sim_time, node_models)
    elif constants.EXPERIMENT == constants.WSCC:
        return wscc.aggregation(aggregation_round, node_id, sim_time, node_models)

def get_participating_nodes(node_id, sim_time):
    logs_data = {'event': 'get_participating_nodes', 'node_id': node_id, 'sim_time': sim_time}
    logs.register_log(logs_data)

    if constants.EXPERIMENT == constants.WSCC:
        return wscc.get_participating_nodes(node_id, sim_time)

def get_cluster_weights(node_id, cluster, sim_time):
    logs_data = {'event': 'get_cluster_weights', 'node_id': node_id, 'cluster': cluster, 'sim_time': sim_time}
    logs.register_log(logs_data)

    if constants.EXPERIMENT == constants.WSCC:
        return wscc.get_cluster_weights(node_id, cluster, sim_time)

def get_cluster_nodes(node_id, cluster, sim_time):
    logs_data = {'event': 'get_cluster_nodes', 'node_id': node_id, 'cluster': cluster, 'sim_time': sim_time}
    logs.register_log(logs_data)

    if constants.EXPERIMENT == constants.WSCC:
        return wscc.get_cluster_nodes(node_id, cluster, sim_time)
