import logging
import numpy as np
from python.src import constants, models, logs

received_weights = {}
dataset_sizes = {}

def store_weights(raw_weights, dataset_size, node_id, sender_id):
    weights = models.decode_weights(raw_weights)
    if node_id not in received_weights.keys():
        received_weights[node_id] = {}
        dataset_sizes[node_id] = {}
    received_weights[node_id][sender_id] = weights
    dataset_sizes[node_id][sender_id] = dataset_size

def aggregation(aggregation_round, node_id, number_of_received_models, sim_time, node_models):
    model = node_models[node_id]

    if node_id in received_weights.keys() and len(received_weights[node_id]) > 0:
        weights = np.zeros(model.get_weights().shape)
        for i in range(len(weights)):
            size = 0
            for sender, rweights in received_weights[node_id].items():
                weights[i] += rweights[i] * dataset_sizes[node_id][sender]
                size += dataset_sizes[node_id][sender]
            weights[i] = weights[i] / size
        model.set_weights(weights)

    logs_data = {'event': 'aggregation', 'node_id': node_id, 'sim_time': sim_time, 'aggregation_round': aggregation_round, 'number_of_received_models': number_of_received_models}
    logs.register_log(logs_data)

    models.save_weights(node_id, model.get_weights())
    node_models[node_id] = model

    received_weights[node_id] = {}
    dataset_sizes[node_id] = {}

def train(node_id, training_round, sim_time, vehicle_data, vehicle_models):
    X_train, y_train = vehicle_data[node_id]['train']
    X_valid, y_valid = vehicle_data[node_id]['valid']

    model = vehicle_models[node_id]
    history = model.fit(X_train, y_train, epochs=constants.EPOCHS, validation_data=(X_valid, y_valid), verbose=0)

    logging.warning('Node {}, Training Round {}, History {}'.format(node_id, training_round, history.history))
    logs_data = {'event': 'train', 'node_id': node_id, 'sim_time': sim_time, 'training_round': training_round, 'history': history.history}
    logs.register_log(logs_data)

    models.save_weights(node_id, model.get_weights())
    vehicle_models[node_id] = model